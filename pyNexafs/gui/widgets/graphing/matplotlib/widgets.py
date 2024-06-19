from collections.abc import Callable
from typing import Any, Collection, Literal
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import _SelectorWidget, SpanSelector, ToolLineHandles
import numpy.typing as npt
import matplotlib.cbook as cbook
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import overrides


class NSpanSelector(SpanSelector):
    """
    Code based on matplotlib.widgets.SpanSelector.

    This class is a modified version of SpanSelector that allows for multiple selections.
    """

    @overrides.overrides
    def __init__(
        self,
        N: int,
        ax: Axes,
        onselect: Callable[[float, float], Any],
        direction: Literal["horizontal"] | Literal["vertical"],
        minspan: float = 0,
        useblit: bool = False,
        props: dict[str, Any] | None = {"facecolor": "red", "alpha": 0.5},
        interactive: bool = False,
        grab_range: float = 10,
        drag_from_anywhere: bool = False,
        ignore_event_outside: bool = False,
        button: MouseButton | Collection[MouseButton] | None = None,
        onmove_callback: Callable[[float, float], Any] | None = None,
        handle_props: dict[str, Any] | None = None,
        state_modifier_keys: dict[str, str] | None = None,
        snap_values: npt.ArrayLike = None,
    ) -> None:
        # Define N
        self.N = N
        super().__init__(
            ax=ax,
            onselect=onselect,
            direction=direction,
            minspan=minspan,
            useblit=useblit,
            props=props,
            interactive=interactive,
            grab_range=grab_range,
            drag_from_anywhere=drag_from_anywhere,
            ignore_event_outside=ignore_event_outside,
            button=button,
            onmove_callback=onmove_callback,
            handle_props=handle_props,
            state_modifier_keys=state_modifier_keys,
            snap_values=snap_values,
        )

    # --------------- Override attributes of _SelectorWidget ---------------:
    @property
    def artists(self):
        # Overrides to include self._selection_artists instead of
        # self._selection_artist in the list of artists.
        handles_artists = getattr(self, "_handles_artists", ())
        return (*self._selection_artists,) + handles_artists

    @overrides.overrides
    def set_props(self, **props) -> None:
        # Overrides to set properties for all selection artists,
        # instead of just the single selection artist.
        artist = self._selection_artists
        for artist in self.artists:
            props = cbook.normalize_kwargs(props, artist)
            artist.set(**props)
        if self.useblit:
            self.update()

    # --------------- Override attributes of SpanSelector ---------------:
    @overrides.overrides
    def new_axes(self, ax, *, _props=None):
        """Set SpanSelector to operate on a new Axes."""
        # Overrides to define self.N rectangles for selection in rect_artists
        # and self._selection_artists variables. Previously was rect_artist and
        # self._selection_artist.
        self.ax = ax
        assert isinstance(self.ax, Axes)
        if self.canvas is not ax.figure.canvas:
            if self.canvas is not None:
                self.disconnect_events()

            self.canvas = ax.figure.canvas
            self.connect_default_events()

        # Reset
        self._selection_completed = False
        # Direction of variables.
        if self.direction == "horizontal":
            trans = ax.get_xaxis_transform()
            w, h = 0, 1
        else:
            trans = ax.get_yaxis_transform()
            w, h = 1, 0

        # Generate rectangles for selection.
        # use double the number of selections to allow spaces between
        dw = w / (2 * self.N)
        dh = h / (2 * self.N)
        rect_artists = [
            Rectangle((2 * i * dw, 2 * i * dh), dw, dh, transform=trans, visible=False)
            for i in range(self.N)
        ]
        for i in range(len(rect_artists)):
            rect_artist = rect_artists[i]
            if _props is not None:
                rect_artist.update(_props)
            elif self._selection_artists is not None and len(
                self._selection_artists
            ) == len(rect_artists):
                rect_artist.update_from(self._selection_artists[i])
            self.ax.add_patch(rect_artist)
        self._selection_artists = rect_artists

    @overrides.override
    def _setup_edge_handles(self, props):
        ## Overrides to define 2*self.N handles for selection in _edge_handles.
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == "horizontal":
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()
        # Update positions to include handles at each Nth position
        dxy = (positions[1] - positions[0]) / (2 * self.N - 1)
        nPositions = (dxy * i for i in range(2 * self.N))
        # Set handles.
        self._edge_handles = ToolLineHandles(
            self.ax,
            nPositions,
            direction=self.direction,
            line_props=props,
            useblit=self.useblit,
        )

    @overrides.overrides
    def _press(self, event):
        """Button press event handler."""
        # Overrides to use self._selection_artists instead of self._selection_artist.
        # Also adjust extents for only the active handle.
        self._set_cursor(True)
        if self._interactive and self._selection_artists.get_visible():
            self._set_active_handle(event)
        else:
            self._active_handle = None

        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        xdata, ydata = self._get_data_coords(event)
        v = xdata if self.direction == "horizontal" else ydata

        if self._active_handle is None and not self.ignore_event_outside:
            # when the press event outside the span, we initially set the
            # visibility to False and extents to (v, v)
            # update will be called when setting the extents
            self._visible = False
            self.extents = v, v
            # We need to set the visibility back, so the span selector will be
            # drawn when necessary (span width > 0)
            self._visible = True
        else:
            self.set_visible(True)

        return False

    @overrides.overrides
    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Overrides to track which handle index is active.

        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        self._active_handle_idx = e_idx

        # Prioritise center handle over other handles
        # Use 'C' to match the notation used in the RectangleSelector
        if "move" in self._state:
            self._active_handle = "C"
        elif e_dist > self.grab_range:
            # Not close to any handles
            self._active_handle = None
            if self.drag_from_anywhere and self._contains(event):
                # Check if we've clicked inside the region
                self._active_handle = "C"
                self._extents_on_press = self.extents
            else:
                self._active_handle = None
                return
        else:
            # Closest to an edge handle
            self._active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents

    @overrides.overrides
    def _release(self, event):
        """Button release event handler."""
        self._set_cursor(False)

        if not self._interactive:
            self._selection_artist.set_visible(False)

        if (
            self._active_handle is None
            and self._selection_completed
            and self.ignore_event_outside
        ):
            return

        ext = self.extents
        spans = [vmax - vmin for vmin, vmax in ext]

        for i, span in enumerate(spans):
            if span <= self.minspan:
                # Remove span and set self._selection_completed = False
                self.set_visible(False)
                if self._selection_completed:
                    # Call onselect, only when the span is already existing
                    self.onselect(vmin, vmax)
                self._selection_completed = False
            else:
                self.onselect(vmin, vmax)
                self._selection_completed = True

        self.update()

        self._active_handle = None

        return False

    @property
    def extents(self) -> list[tuple[float, float]]:
        """
        Returns
        -------
        list[tuple[float, float]]
            The values, in data coordinates, for the start and end points of all current rectangles.
            If there is no selection then the start and end values will be the same.
        """
        # Overrides to use self._selection_artists instead of self._selection_artist.
        vals = []
        for selection_artist in self._selection_artists:
            if self.direction == "horizontal":
                vmin = selection_artist.get_x()
                vmax = vmin + selection_artist.get_width()
            else:
                vmin = selection_artist.get_y()
                vmax = vmin + selection_artist.get_height()
            vals.append((vmin, vmax))
        return vals

    @extents.setter
    def extents(self, extents: list[tuple[float, float]]):
        # Update displayed shape
        if self.snap_values is not None:
            extents = tuple(self._snap(extents, self.snap_values))
        self._draw_shape(*extents)
        if self._interactive:
            # Update displayed handles
            self._edge_handles.set_data(self.extents)
        self.set_visible(self._visible)
        self.update()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    mpl.use("Qt5Agg")
    fig, ax = plt.subplots()
    ax.plot(np.sin(np.linspace(0, 10, 100)))

    def onselect(xmin, xmax):
        print(xmin, xmax)

    span = NSpanSelector(
        2, ax, onselect, "horizontal", interactive=True, useblit=True, interactive=True
    )
    plt.show()
