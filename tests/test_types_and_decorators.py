"""Tests for pyNexafs.types.dtype and pyNexafs.utils.decorators.enum_member_doc."""

import typing
from enum import StrEnum

import pytest

from pyNexafs.types import dtype, ALL_DTYPE_MEMBERS
from pyNexafs.utils.decorators import enum_member_doc


##############################################################################
###############  enum_member_doc — trailing string literal  ##################
##############################################################################


class TestEnumMemberDocLiteral:
    """enum_member_doc correctly picks up trailing string literals."""

    @enum_member_doc
    class _Literal(StrEnum):
        RED = "red"
        """The colour red."""
        GREEN = "green"
        """The colour green."""
        UNDOCUMENTED = "undocumented"

    def test_doc_set_from_literal(self):
        assert self._Literal.RED.__doc__ == "The colour red."
        assert self._Literal.GREEN.__doc__ == "The colour green."

    def test_undocumented_member_doc_unchanged(self):
        # Members without a trailing literal should retain whatever __doc__
        # the enum machinery sets (None or a default string); crucially it
        # must NOT be one of the other members' docstrings.
        doc = self._Literal.UNDOCUMENTED.__doc__
        assert doc != "The colour red."
        assert doc != "The colour green."

    def test_value_unchanged(self):
        assert self._Literal.RED.value == "red"
        assert self._Literal.GREEN.value == "green"

    def test_member_count_unchanged(self):
        assert len(self._Literal) == 3


##############################################################################
###########  enum_member_doc — Annotated[T, "doc"] metadata  ################
##############################################################################


class TestEnumMemberDocAnnotated:
    """enum_member_doc correctly reads Annotated metadata from source annotations."""

    # NOTE: Annotated annotations on enum members cause Pylance
    # reportGeneralTypeIssues, so this style is tested here for correctness
    # but the dtype class uses trailing literals instead.

    def test_doc_from_annotated(self):
        # Defined inside the test so the source file is readable by inspect.
        @enum_member_doc
        class _Ann(StrEnum):
            BLUE: typing.Annotated[str, "The colour blue."] = "blue"

        assert _Ann.BLUE.__doc__ == "The colour blue."

    def test_annotated_overrides_literal(self):
        @enum_member_doc
        class _Override(StrEnum):
            BLUE: typing.Annotated[str, "Annotated wins."] = "blue"
            """Trailing literal loses."""

        assert _Override.BLUE.__doc__ == "Annotated wins."

    def test_value_unchanged_annotated(self):
        @enum_member_doc
        class _Ann(StrEnum):
            BLUE: typing.Annotated[str, "The colour blue."] = "blue"

        assert _Ann.BLUE.value == "blue"


##############################################################################
###########  enum_member_doc — Annotated written to __annotations__  ########
##############################################################################


class TestEnumMemberDocAnnotationsWriteback:
    """After decoration every documented member has an Annotated annotation."""

    @enum_member_doc
    class _Writeback(StrEnum):
        RED = "red"
        """The colour red."""
        UNDOCUMENTED = "undocumented"

    def test_annotated_present_for_documented_member(self):
        hints = typing.get_type_hints(self._Writeback, include_extras=True)
        assert "RED" in hints
        assert typing.get_origin(hints["RED"]) is typing.Annotated

    def test_annotated_doc_matches_member_doc(self):
        hints = typing.get_type_hints(self._Writeback, include_extras=True)
        args = typing.get_args(hints["RED"])
        doc_from_hint = next((m for m in args[1:] if isinstance(m, str)), None)
        assert doc_from_hint == self._Writeback.RED.__doc__

    def test_undocumented_member_not_annotated(self):
        hints = typing.get_type_hints(self._Writeback, include_extras=True)
        if "UNDOCUMENTED" in hints:
            # If present, must not carry a str doc in its metadata
            origin = typing.get_origin(hints["UNDOCUMENTED"])
            if origin is typing.Annotated:
                args = typing.get_args(hints["UNDOCUMENTED"])
                assert not any(isinstance(m, str) for m in args[1:])

    def test_existing_annotated_doc_replaced(self):
        @enum_member_doc
        class _Existing(StrEnum):
            BLUE: typing.Annotated[str, "Original."] = "blue"

        hints = typing.get_type_hints(_Existing, include_extras=True)
        args = typing.get_args(hints["BLUE"])
        doc_from_hint = next((m for m in args[1:] if isinstance(m, str)), None)
        assert doc_from_hint == "Original."

    def test_extra_annotated_metadata_preserved(self):
        sentinel = object()

        @enum_member_doc
        class _Extra(StrEnum):
            BLUE: typing.Annotated[str, "Doc.", sentinel] = "blue"  # type: ignore[misc]

        hints = typing.get_type_hints(_Extra, include_extras=True)
        args = typing.get_args(hints["BLUE"])
        assert sentinel in args


##############################################################################
###########  enum_member_doc — no source available (OSError path)  ##########
##############################################################################


class TestEnumMemberDocNoSource:
    """Decorator degrades gracefully when inspect.getsource raises OSError."""

    def test_annotated_still_works_without_source(self, monkeypatch):
        import inspect as _inspect
        import pyNexafs.utils.decorators as _dec_mod

        _ = _inspect.getsource

        def _raise(*args, **kwargs):
            raise OSError("source code not available")

        monkeypatch.setattr(_dec_mod.inspect, "getsource", _raise)

        @enum_member_doc
        class _NoSrc(StrEnum):
            BLUE: typing.Annotated[str, "Blue without source."] = "blue"

        assert _NoSrc.BLUE.__doc__ == "Blue without source."

    def test_literal_silently_skipped_without_source(self, monkeypatch):
        import pyNexafs.utils.decorators as _dec_mod

        def _raise(*args, **kwargs):
            raise OSError("source code not available")

        monkeypatch.setattr(_dec_mod.inspect, "getsource", _raise)

        # Without source the trailing literal cannot be parsed; __doc__ is
        # whatever the enum machinery sets.
        @enum_member_doc
        class _NoSrc(StrEnum):
            RED = "red"
            """Docstring that cannot be found."""

        assert _NoSrc.RED.__doc__ != "Docstring that cannot be found."


##############################################################################
##########################  dtype members  ###################################
##############################################################################


class TestDtypeValues:
    """dtype enum members have the expected string values."""

    def test_all_members_included(self):
        expected = set(ALL_DTYPE_MEMBERS)
        found = set()
        members = set(dtype.__members__.values())
        for exp in expected:
            assert exp in members, f"{exp} not found in dtype members"
            found.add(exp)
        expected -= found
        assert not expected, f"Expected members not found: {expected}"

    @pytest.mark.parametrize("member", ALL_DTYPE_MEMBERS)
    def test_is_str(self, member):
        assert isinstance(member, str)


class TestDtypeDocs:
    """Every dtype member has a non-empty __doc__ string."""

    @pytest.mark.parametrize("member", ALL_DTYPE_MEMBERS)
    def test_doc_is_set(self, member):
        assert member.__doc__ is not None
        assert isinstance(member.__doc__, str)
        assert len(member.__doc__.strip()) > 0

    def test_docs_are_unique(self):
        docs = [m.__doc__.strip() for m in ALL_DTYPE_MEMBERS]
        assert len(docs) == len(set(docs)), "Duplicate docstrings found: " + str(
            [d for d in docs if docs.count(d) > 1]
        )


class TestDtypeAnnotations:
    """dtype members have Annotated annotations written back after decoration."""

    def test_all_members_have_annotated_hint(self):
        hints = typing.get_type_hints(dtype, include_extras=True)
        for member in ALL_DTYPE_MEMBERS:
            assert member.name in hints, f"{member.name} not in hints"
            assert typing.get_origin(hints[member.name]) is typing.Annotated

    @pytest.mark.parametrize("member", ALL_DTYPE_MEMBERS)
    def test_annotated_doc_matches_member_doc(self, member):
        hints = typing.get_type_hints(dtype, include_extras=True)
        args = typing.get_args(hints[member.name])
        doc_from_hint = next((m for m in args[1:] if isinstance(m, str)), None)
        assert doc_from_hint is not None
        assert doc_from_hint == member.__doc__


##############################################################################
##########################  dtype module aliases  ############################
##############################################################################


class TestDtypeAliases:
    """Module-level aliases in types.py point to the correct dtype members."""

    def test_aliases(self):
        from pyNexafs import types

        for member in ALL_DTYPE_MEMBERS:
            assert getattr(types, member.name) is member
