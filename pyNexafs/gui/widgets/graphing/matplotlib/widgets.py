import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Collection, Literal, Callable
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import SpanSelector, ToolLineHandles  # , _SelectorWidget
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
        onselect: list[Callable[[float, float], Any]] | Callable[[float, float], Any],
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
        # Initialise tracking for active handle.
        self._active_handle_idx = None
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

    from matplotlib.colors import Colormap
    from matplotlib.backend_bases import Event as mplEvent

    @overrides.overrides
    def update_background(
        self, event: mplEvent, cmap: Colormap = mpl.colormaps.get("tab10")
    ) -> None:
        super().update_background(event)
        # Change colour of each selection artist using a cmap
        for i, selection_artist in enumerate(self._selection_artists):
            selection_artist.set_facecolor(cmap(i))
        return

    # --------------- Override attributes of SpanSelector ---------------:
    @overrides.overrides
    def _setup_edge_handles(self, props):
        # Define initial position using the axis bounds to keep the same bounds
        if self.direction == "horizontal":
            positions = self.ax.get_xbound()
        else:
            positions = self.ax.get_ybound()

        # Use N to define 2*N handles for selection in _edge_handles.
        dxy = (positions[1] - positions[0]) / (2 * self.N - 1)
        nPositions = (dxy * i for i in range(2 * self.N))

        self._edge_handles = ToolLineHandles(
            self.ax,
            nPositions,
            direction=self.direction,
            line_props=props,
            useblit=self.useblit,
        )

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

        dw = w / (2 * self.N - 1)
        dh = h / (2 * self.N - 1)
        rect_artists = [
            Rectangle(
                xy=(
                    (2 * i * dw, 0)
                    if self.direction == "horizontal"
                    else (0, 2 * i * dh)
                ),
                width=w,
                height=h,
                transform=trans,
                visible=False,
            )
            for i in range(self.N)
        ]
        for i, rect_artist in enumerate(rect_artists):
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
        nPositions = (positions[0] + dxy * i for i in range(2 * self.N))
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
        if self._interactive and any(
            [select_artist.get_visible() for select_artist in self._selection_artists]
        ):
            self._set_active_handle(event)
        else:
            self._active_handle = None
            self._active_handle_idx = None

        if self._active_handle is None or not self._interactive:
            # Clear previous rectangle before drawing new rectangle.
            self.update()

        xdata, ydata = self._get_data_coords(event)
        v = xdata if self.direction == "horizontal" else ydata

        if self._active_handle is None and not self.ignore_event_outside:
            # Collect closest edge handle to the mouse press event.
            closest, dist = self._edge_handles.closest(xdata, ydata)
            handle_idx = int(np.floor(closest / 2))
            minmax = closest % 2

            # when the press event outside the span, we initially set the
            # visibility to False and extents to (v, v)
            # update will be called when setting the extents
            self.set_visible(False, closest)
            ex = self.extents
            ex[handle_idx] = v, v
            self.extents = ex
            # We need to set the visibility back, so the span selector will be
            # drawn when necessary (span width > 0)
            self.set_visible(True, closest)
        else:
            self.set_visible(True)

        return False

    @overrides.overrides
    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event."""
        # Overrides to track which handle index is active.

        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)
        h_idx = int(np.floor(e_idx / 2))  # Handle index
        edge_order_idx = e_idx % 2  # 0 for min, 1 for max

        # Save selection index. 2 handles per selection, so divide by 2.
        self._active_handle_idx = h_idx
        # Save coordinates of rectangle at the start of handle movement.
        self._extents_on_press = self.extents[h_idx]

        ## Prioritise within proximity to edge handle, then centre handle, then outside.
        if e_dist < self.grab_range:
            # Closest to an edge handle
            self._active_handle = self._edge_order[edge_order_idx]
        elif self.drag_from_anywhere and self._contains(event):
            # Check if we've clicked inside any region.
            # Note: self._contains changes self._active_handle_idx if True for a different index.
            # Update the extents, in case the active handle index has changed.
            self._extents_on_press = self.extents[self._active_handle_idx]
            self._active_handle = "C"
        elif "move" in self._state:
            self._active_handle = "C"
        else:
            # If outside the region, instead use closest edge.
            self._active_handle = self._edge_order[edge_order_idx]

    @overrides.overrides
    def _contains(self, event):
        """Return True if event is within the selected handle index."""
        i = self._active_handle_idx
        if i is None:
            return False
        else:
            if self._selection_artists[i].contains(event, radius=0)[0]:
                return True
            else:
                # Loop through other selection artists to check if event is within any.
                for j, selection_artist in enumerate(self._selection_artists):
                    if j != i and selection_artist.contains(event, radius=0)[0]:
                        # Update active handle index to the selection artist that contains the event.
                        self._active_handle_idx = j
                        return True

    @overrides.overrides
    def _onmove(self, event):
        """Motion notify event handler."""

        xdata, ydata = self._get_data_coords(event)
        if self.direction == "horizontal":
            v = xdata
            vpress = self._eventpress.xdata
        else:
            v = ydata
            vpress = self._eventpress.ydata

        # move existing span
        # When "dragging from anywhere", `self._active_handle` is set to 'C'
        # (match notation used in the RectangleSelector)
        if self._active_handle == "C" and self._extents_on_press is not None:
            vmin, vmax = self._extents_on_press
            dv = v - vpress
            vmin += dv
            vmax += dv

        # resize an existing shape
        elif (
            self._active_handle
            and self._active_handle != "C"
            and self._extents_on_press is not None
        ):
            vmin, vmax = self._extents_on_press
            if self._active_handle == "min":
                vmin = v
            else:
                vmax = v
        # new shape
        else:
            # Don't create a new span if there is already one when
            # ignore_event_outside=True
            if self.ignore_event_outside and self._selection_completed:
                return
            vmin, vmax = vpress, v
            if vmin > vmax:
                vmin, vmax = vmax, vmin

        if self._active_handle_idx is not None:
            ex = self.extents
            ex[self._active_handle_idx] = vmin, vmax
            self.extents = ex

        if self.onmove_callback is not None:
            self.onmove_callback(vmin, vmax)

        return False

    @overrides.overrides
    def set_visible(self, visible: bool, index: int = None) -> None:
        """
        Overrides functionality of default _SelectorWidget.set_visible method.

        Parameters
        ----------
        visible : bool
            The visibility of the selection artists.
        index : int, optional
            The index of the selection artist to set visibility for. If None, then
            all selection artists will be set to the same visibility.
        """
        if index is not None:
            self.artists[index].set_visible(visible)
        else:
            self._visible = visible
            for artist in self.artists:
                artist.set_visible(visible)

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
            vmin, vmax = ext[i]
            if span <= self.minspan:
                # Remove span and set self._selection_completed = False
                self.set_visible(False, index=i)
                if self._selection_completed:
                    # Call onselect, only when the span is already existing
                    if (
                        isinstance(self.onselect, list)
                        and self._active_handle_idx is not None
                    ):
                        # Call onselect, only when the handle index matches
                        if i == self._active_handle_idx:
                            self.onselect[i](vmin, vmax)
                    else:
                        self.onselect(vmin, vmax)
                self._selection_completed = False
            else:
                if (
                    isinstance(self.onselect, list)
                    and self._active_handle_idx is not None
                ):
                    # Call onselect, only when the handle index matches
                    if i == self._active_handle_idx:
                        self.onselect[i](vmin, vmax)
                else:
                    self.onselect(vmin, vmax)
                self._selection_completed = True

        self.update()

        # Reset active handle
        self._active_handle = None
        self._active_handle_idx = None

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
        if len(extents) > self.N:
            raise ValueError("Number of extents must be equal to or less than N.")
        # Update displayed shape
        if self.snap_values is not None:
            for i, extent in enumerate(extents):
                extents[i] = tuple(self._snap(extent, self.snap_values))
        self._draw_shapes(extents)
        if self._interactive:
            # Update displayed handles
            extent_data = np.array(extents).flatten()
            self._edge_handles.set_data(extent_data)
        self.set_visible(self._visible)
        self.update()

    def _draw_shapes(
        self,
        extents: list[tuple[float, float]],
    ):
        """An alternative method for SpanSelector._draw_shape.
        Draws the selection shapes on the axes."""
        for i, extent in enumerate(extents):
            (vmin, vmax) = extent
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            if self.direction == "horizontal":
                self._selection_artists[i].set_x(vmin)
                self._selection_artists[i].set_width(vmax - vmin)
            else:
                self._selection_artists[i].set_y(vmin)
                self._selection_artists[i].set_height(vmax - vmin)

    @property
    def colors_rect(self) -> list[str]:
        """
        Property to get/set the colours of the rectangle spans.

        Parameters
        ----------
        colors : list[str]
            List of colours to set for each rectangle span.

        Returns
        -------
        list[str]
            List of colours for each rectangle
        """
        return [artist.get_facecolor() for artist in self._selection_artists]

    @colors_rect.setter
    def colors_rect(self, colors: list[str] | str):
        if isinstance(colors, list):
            for i, color in enumerate(colors):
                self._selection_artists[i].set_facecolor(color)
        elif isinstance(colors, str):
            for artist in self._selection_artists:
                artist.set_facecolor(colors)
        else:
            raise ValueError(f"colors '{colors}' must be a list or a string")
        self.update()

    @property
    def colors_edges(self) -> list[tuple[str, str]]:
        """
        Property to get the colours of the edge handles.

        Parameters
        ----------
        colors : list[tuple[str,str]]
            List of colours to set for each edge handle.

        Returns
        -------
        list[tuple[str,str]]
            List of colours for each edge handle
        """
        colors = [artist.get_color() for artist in self._edge_handles.artists]
        colors = [(colors[i], colors[i + 1]) for i in range(0, len(colors), 2)]
        return colors

    @colors_edges.setter
    def colors_edges(self, colors: list[tuple[str, str]] | tuple[str, str] | str):
        if isinstance(colors, list):
            for i, color in enumerate(colors):
                self._edge_handles.artists[i * 2].set_color(color[0])
                self._edge_handles.artists[i * 2 + 1].set_color(color[1])
        if isinstance(colors, tuple):
            for i, artist in enumerate(self._edge_handles.artists):
                if i % 2 == 0:
                    artist.set_color(colors[0])
                else:
                    artist.set_color(colors[1])
        elif isinstance(colors, str):
            for artist in self._edge_handles.artists:
                artist.set_color(colors)
        else:
            raise ValueError(
                f"colors {colors} must be a list of tuples, a tuple or a string"
            )
        self.update()


if __name__ == "__main__":

    mpl.use("QtAgg")
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)

    span = 5

    def onselect1(xmin, xmax):
        print("1", xmin, xmax)
        pass

    def onselect2(xmin, xmax):
        print("2", xmin, xmax)
        pass

    def onselect3(xmin, xmax):
        print("3", xmin, xmax)
        thecols = span.colors_rect[-1]
        print(thecols)
        # thecols[-1]
        # span.colors_rect = thecols
        pass

    span = NSpanSelector(
        N=3,
        ax=ax,
        onselect=[onselect1, onselect2, onselect3],
        direction="horizontal",
        drag_from_anywhere=True,
        interactive=True,
        useblit=True,
    )
    span.extents = [(1, 2), (3, 4), (5, 6)]

    plt.show()
